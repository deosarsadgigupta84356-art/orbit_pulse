"""Historical ESA dataset intelligence dashboard."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.esa_data import load_esa_test_data
from src.esa_predict import ESATwoStagePredictor
from src.esa_features import build_esa_event_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@st.cache_data(show_spinner="Loading public ESA CDM inputs...")
def load_public_inputs():
    return load_esa_test_data()


@st.cache_resource
def load_esa_model():
    return ESATwoStagePredictor()


@st.cache_data(show_spinner="Generating historical ESA event predictions...")
def generate_esa_predictions():
    inputs = load_public_inputs()
    predictor = load_esa_model()
    return predictor.predict(inputs)


def load_json_report(relative_path):
    path = PROJECT_ROOT / relative_path
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
@st.cache_data(show_spinner="Preparing ESA intelligence workspace...")
def load_esa_workspace():
    inputs = load_public_inputs()
    predictions = generate_esa_predictions()
    metadata = load_json_report("models/esa/esa_model_metadata.json")
    metrics = load_json_report("reports/esa/esa_validation_metrics.json")

    features = build_esa_event_features(inputs)

    return {
        "inputs": inputs,
        "predictions": predictions,
        "metadata": metadata,
        "metrics": metrics,
        "features": features,
    }

def select_esa_event(predictions, key):
    event_ids = sorted(
        predictions["event_id"].astype(str).unique()
    )

    search_text = st.text_input(
        "Search Event ID",
        placeholder="Type an event ID to filter...",
        key=f"{key}_search",
    )

    filtered_ids = [
        event_id
        for event_id in event_ids
        if search_text.strip().lower() in event_id.lower()
    ]

    if not filtered_ids:
        st.warning("No matching event IDs found. Try another search.")
        return None

    selected_event = st.selectbox(
        "Select Event ID",
        filtered_ids,
        key=f"{key}_select",
    )

    st.caption(
        f"Selected Event ID: {selected_event} | "
        f"{len(filtered_ids):,} matching events out of {len(event_ids):,}"
    )

    return selected_event


def render_esa_mission_overview(workspace):
    inputs = workspace["inputs"]
    predictions = workspace["predictions"]
    metadata = workspace["metadata"]

    st.title("Mission Overview")
    st.caption(
        "Historical ESA conjunction intelligence | "
        "ESA Collision Avoidance Challenge"
    )

    st.info(
        "HISTORICAL OFFLINE INFERENCE - Public historical ESA CDM inputs."
        "This is not a live operational feed and does not authorize "
        "autonomous spacecraft manoeuvres."
    )

    floor_count = int(predictions["predicted_risk_floor"].sum())
    above_floor_count = len(predictions) - floor_count
    eligible_events = predictions["event_id"].nunique()

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Public CDM Records", f"{len(inputs):,}")
    metric_2.metric("Eligible Events", f"{eligible_events:,}")
    metric_3.metric("Predicted Risk-Floor", f"{floor_count:,}")
    metric_4.metric("Predicted Above-Floor", f"{above_floor_count:,}")

    st.caption(
        "Predictions are event-level estimates generated from eligible "
        "CDMs at least 48 hours before TCA. The -30 risk floor is not "
        "zero collision probability or proof of safety."
    )

    st.divider()
    st.subheader("Predicted Event-Level Risk Distribution")

    risk_values = predictions["predicted_final_risk"].dropna()
    if not risk_values.empty:
        histogram = pd.DataFrame({
            "Predicted events": risk_values.value_counts(
                bins=12, sort=False
            )
        })
        st.bar_chart(histogram)
    else:
        st.warning("No event-level risk predictions are available.")

    st.divider()
    st.subheader("Risk-Floor Classification")

    classification = pd.DataFrame({
        "Prediction category": ["Predicted risk floor", "Predicted above floor"],
        "Events": [floor_count, above_floor_count],
    }).set_index("Prediction category")

    st.bar_chart(classification)

    st.divider()
    st.subheader("Highest Predicted Event Risks")

    display_columns = [
        "event_id",
        "predicted_final_risk",
        "predicted_risk_floor",
        "selected_time_to_tca_days",
        "prediction_cutoff_days",
    ]

    st.dataframe(
        predictions[display_columns].sort_values(
            "predicted_final_risk",
            ascending=False,
        ).head(20),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        f"Model: {metadata['model_name']} | "
        f"Prediction boundary: TCA - "
        f"{metadata['prediction_cutoff_days']} days | "
        "Historical offline inference"
    )
def render_esa_risk_monitor(workspace):
    inputs = workspace["inputs"]
    predictions = workspace["predictions"]

    st.title("Historical Risk Monitor")
    st.caption(
        "ESA historical CDM activity | Event-level offline inference"
    )

    st.info(
        "HISTORICAL MONITOR - This view explores archived ESA CDM "             "sequences. It is not a live telemetry or operational feed."
    )

    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Historical CDM Records", f"{len(inputs):,}")
    metric_2.metric("Conjunction Events", f"{inputs['event_id'].nunique():,}")
    metric_3.metric(
        "Average CDMs per Event",
        f"{len(inputs) / inputs['event_id'].nunique():.2f}",
    )

    st.divider()
    st.subheader("Historical Event Activity")

    activity = (
        inputs.groupby("time_to_tca")
        .size()
        .rename("CDM records")
        .reset_index()
        .sort_values("time_to_tca", ascending=False)
        .set_index("time_to_tca")
    )

    st.line_chart(activity)
    st.caption(
        "Number of archived CDM records by time-to-TCA. "
        "All public test observations satisfy the 48-hour cutoff."
    )

    st.divider()
    st.subheader("Inspect Individual Event History")

    selected_event = select_esa_event(
        predictions,
        "esa_monitor_event",
    )
    if selected_event is None:
        return

    event_history = inputs[
        inputs["event_id"].astype(str) == selected_event
    ].sort_values("time_to_tca", ascending=False)

    prediction_row = predictions[
        predictions["event_id"].astype(str) == selected_event
    ].iloc[0]

    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("CDM Observations", f"{len(event_history):,}")
    metric_2.metric(
        "Earliest Available TCA Window",
        f"{event_history['time_to_tca'].max():.2f} days",
    )
    metric_3.metric(
        "Latest Available TCA Window",
        f"{event_history['time_to_tca'].min():.2f} days",
    )

    st.markdown("**Model-predicted final event-level risk**")
    st.metric(
        "Predicted base-10 logarithmic risk",
        f"{prediction_row['predicted_final_risk']:.4f}",
    )

    st.caption(
        "This is a model prediction, not an observed final event outcome. "
        f"Risk-floor classification: "
        f"{'Predicted risk floor' if prediction_row['predicted_risk_floor'] else 'Predicted above floor'}."
    )

    st.divider()
    st.subheader("Observed Historical CDM Risk Sequence")

    risk_history = (
        event_history[["time_to_tca", "risk"]]
        .set_index("time_to_tca")
        .sort_index(ascending=False)
    )

    st.line_chart(risk_history)

    st.caption(
        "The historical CDM risk field is an input observation from each "
        "message. It is distinct from the model-predicted final event-level "
        "risk shown above."
    )

    st.divider()
    st.subheader("CDM Observation Details")

    display_columns = [
        "time_to_tca",
        "risk",
        "max_risk_estimate",
        "miss_distance",
        "relative_speed",
        "mahalanobis_distance",
    ]

    st.dataframe(
        event_history[display_columns],
        use_container_width=True,
        hide_index=True,
    )
def render_esa_event_explorer(workspace):
    predictions = workspace["predictions"]
    features = workspace["features"].reset_index()

    st.title("Event Explorer")
    st.caption(
        "Search, filter and compare historical ESA conjunction events"
    )

    event_register = predictions.merge(
        features[
            [
                "event_id",
                "c_object_type",
                "miss_distance",
                "relative_speed",
                "mahalanobis_distance",
            ]
        ],
        on="event_id",
        how="left",
        validate="one_to_one",
    )

    st.info(
        "The register combines model predictions with the latest "
        "eligible CDM features selected at least 48 hours before TCA."
    )

    col_1, col_2 = st.columns(2)

    with col_1:
        search_id = st.text_input(
            "Search Event ID",
            key="esa_explorer_search",
        )

    with col_2:
        risk_filter = st.selectbox(
            "Predicted risk classification",
            ["All", "Predicted risk floor", "Predicted above floor"],
            key="esa_explorer_risk",
        )

    object_types = ["All"] + sorted(
        event_register["c_object_type"].dropna().astype(str).unique()
    )

    object_filter = st.selectbox(
        "Chaser object type",
        object_types,
        key="esa_explorer_object",
    )

    filtered = event_register.copy()

    if search_id.strip():
        filtered = filtered[
            filtered["event_id"].astype(str).str.contains(
                search_id.strip(),
                case=False,
                na=False,
            )
        ]

    if risk_filter == "Predicted risk floor":
        filtered = filtered[filtered["predicted_risk_floor"] == True]
    elif risk_filter == "Predicted above floor":
        filtered = filtered[filtered["predicted_risk_floor"] == False]

    if object_filter != "All":
        filtered = filtered[
            filtered["c_object_type"].astype(str) == object_filter
        ]

    st.divider()

    metric_1, metric_2 = st.columns(2)
    metric_1.metric("Matching Events", f"{len(filtered):,}")
    metric_2.metric(
        "Above-Floor Predictions",
        f"{int((~filtered['predicted_risk_floor']).sum()):,}",
    )

    display_columns = [
        "event_id",
        "predicted_final_risk",
        "predicted_risk_floor",
        "c_object_type",
        "miss_distance",
        "relative_speed",
        "mahalanobis_distance",
        "selected_time_to_tca_days",
    ]

    st.dataframe(
        filtered[display_columns].sort_values(
            "predicted_final_risk",
            ascending=False,
        ),
        use_container_width=True,
        hide_index=True,
    )

    csv_data = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Filtered Event Register",
        data=csv_data,
        file_name="orbit_pulse_esa_filtered_events.csv",
        mime="text/csv",
    )

    st.caption(
        "Predicted risk is expressed in base-10 logarithmic units. "
        "It is not directly interchangeable with synthetic collision probability."
    )
def render_esa_event_detail(workspace):
    inputs = workspace["inputs"]
    predictions = workspace["predictions"]
    features = workspace["features"].reset_index()

    st.title("Event Detail")
    st.caption(
        "Individual conjunction investigation | Historical ESA CDM data"
    )

    selected_event = select_esa_event(
        predictions,
        "esa_detail_event",
    )
    if selected_event is None:
        return

    event_history = inputs[
        inputs["event_id"].astype(str) == selected_event
    ].sort_values("time_to_tca", ascending=True)

    prediction = predictions[
        predictions["event_id"].astype(str) == selected_event
    ].iloc[0]

    feature_match = features[
        features["event_id"].astype(str) == selected_event
    ]

    if feature_match.empty:
        st.error("No eligible feature record found for this event.")
        return

    selected_features = feature_match.iloc[0]

    st.info(
        "Historical offline inference. The model uses the latest eligible "
        "CDM at or beyond the 48-hour prediction boundary."
    )

    st.subheader("Event Prediction")

    col_1, col_2, col_3 = st.columns(3)

    col_1.metric(
        "Predicted Final Log Risk",
        f"{prediction['predicted_final_risk']:.4f}",
    )

    col_2.metric(
        "Risk-Floor Classification",
        "Predicted floor" if prediction["predicted_risk_floor"]
        else "Predicted above floor",
    )

    col_3.metric(
        "Selected CDM Time to TCA",
        f"{prediction['selected_time_to_tca_days']:.3f} days",
    )

    st.divider()
    st.subheader("Historical CDM Risk Progression")

    risk_history = (
        event_history[["time_to_tca", "risk"]]
        .set_index("time_to_tca")
        .sort_index()
    )

    st.line_chart(risk_history)

    st.caption(
        "Observed per-CDM risk input values. These are not the model's "
        "predicted final event-level risk."
    )

    st.divider()
    st.subheader("Encounter Geometry")

    col_1, col_2, col_3 = st.columns(3)

    col_1.metric(
        "Miss Distance",
        f"{selected_features['miss_distance']:,.2f} m",
    )

    col_2.metric(
        "Relative Speed",
        f"{selected_features['relative_speed']:,.2f} m/s",
    )

    col_3.metric(
        "Mahalanobis Distance",
        f"{selected_features['mahalanobis_distance']:,.3f}",
    )

    st.markdown("**Relative Position — R / T / N (m)**")

    position = pd.DataFrame({
        "Axis": ["Radial (R)", "Transverse (T)", "Normal (N)"],
        "Relative position (m)": [
            selected_features["relative_position_r"],
            selected_features["relative_position_t"],
            selected_features["relative_position_n"],
        ],
    }).set_index("Axis")

    st.bar_chart(position)

    st.divider()
    st.subheader("Selected Eligible CDM Features")

    feature_columns = [
        "time_to_tca",
        "risk",
        "max_risk_estimate",
        "max_risk_scaling",
        "miss_distance",
        "relative_speed",
        "mahalanobis_distance",
        "relative_position_r",
        "relative_position_t",
        "relative_position_n",
        "relative_velocity_r",
        "relative_velocity_t",
        "relative_velocity_n",
        "t_j2k_sma",
        "t_j2k_ecc",
        "t_j2k_inc",
        "c_object_type",
    ]

    available_columns = [
        column for column in feature_columns
        if column in selected_features.index
    ]

    details = pd.DataFrame({
        "Feature": available_columns,
        "Selected CDM value": [
            selected_features[column]
            for column in available_columns
        ],
    })

    st.dataframe(
        details,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader("Complete Historical CDM Sequence")

    history_columns = [
        "time_to_tca",
        "risk",
        "max_risk_estimate",
        "miss_distance",
        "relative_speed",
        "mahalanobis_distance",
    ]

    st.dataframe(
        event_history[history_columns],
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download Event CDM History",
        data=event_history.to_csv(index=False).encode("utf-8"),
        file_name=f"orbit_pulse_esa_event_{selected_event}.csv",
        mime="text/csv",
    )
def render_esa_evidence(workspace):
    inputs = workspace["inputs"]
    predictions = workspace["predictions"]
    features = workspace["features"].reset_index()

    st.title("Evidence & Explanation")
    st.caption(
        "ESA model input evidence | Two-stage event-level risk inference"
    )

    selected_event = select_esa_event(
        predictions,
        "esa_evidence_event",
    )
    if selected_event is None:
        return

    prediction = predictions[
        predictions["event_id"].astype(str) == selected_event
    ].iloc[0]

    feature_match = features[
        features["event_id"].astype(str) == selected_event
    ]

    if feature_match.empty:
        st.error("No eligible model input record found.")
        return

    evidence = feature_match.iloc[0]
    covariance_columns = [
        "t_sigma_r",
        "t_sigma_t",
        "t_sigma_n",
        "c_sigma_r",
        "c_sigma_t",
        "c_sigma_n",
        "t_sigma_rdot",
        "t_sigma_tdot",
        "t_sigma_ndot",
        "c_sigma_rdot",
        "c_sigma_tdot",
        "c_sigma_ndot",
    ]

    covariance_rows = [
        {
            "Feature": column,
            "Selected input value": evidence[column],
        }
        for column in covariance_columns
        if column in evidence.index
    ]

    st.dataframe(
        pd.DataFrame(covariance_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Position covariance determinants are represented in log10 form "
        "in the model feature matrix to maintain numerical stability. "
        "Sigma fields retain their dataset units."
    )

    st.divider()
    st.subheader("Observation Quality Evidence")

    quality_columns = [
        "t_obs_available",
        "t_obs_used",
        "t_residuals_accepted",
        "t_weighted_rms",
        "t_recommended_od_span",
        "t_actual_od_span",
        "c_obs_available",
        "c_obs_used",
        "c_residuals_accepted",
        "c_weighted_rms",
        "c_recommended_od_span",
        "c_actual_od_span",
    ]

    quality_rows = [
        {
            "Feature": column,
            "Selected input value": evidence[column],
        }
        for column in quality_columns
        if column in evidence.index
    ]

    st.dataframe(
        pd.DataFrame(quality_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader("Missing Input Evidence")

    selected_input = inputs[
        (inputs["event_id"].astype(str) == selected_event)
        & np.isclose(
            inputs["time_to_tca"],
            float(evidence["time_to_tca"]),
        )
    ]

    if not selected_input.empty:
        missing_count = int(selected_input.iloc[0].isna().sum())
        st.metric(
            "Missing fields in selected source CDM",
            f"{missing_count} / {len(inputs.columns)}",
        )

        missing_fields = (
            selected_input.iloc[0][selected_input.iloc[0].isna()]
            .index.tolist()
        )

        if missing_fields:
            st.write("Missing fields:", ", ".join(missing_fields))
        else:
            st.success("No missing fields in this selected source CDM.")

    st.caption(
        "The trained preprocessing pipeline handles missing values. "
        "Missing source values are not silently presented as observed measurements."
    )
def render_esa_model_performance(workspace):
    metrics = workspace["metrics"]
    metadata = workspace["metadata"]

    classifier = metrics["classifier"]["random_forest"]
    classifier_baseline = metrics["classifier"]["majority_class_baseline"]

    regression = metrics["conditional_regression"]
    regression_model = regression["random_forest"]
    median_baseline = regression["median_baseline"]
    observed_baseline = regression["latest_observed_risk_baseline"]

    combined = metrics["combined_prediction"]
    combined_model = combined["two_stage_model"]
    combined_baseline = combined["latest_observed_risk_baseline"]

    st.title("Model Performance & Ablations")
    st.caption("ESA historical event-level validation")

    st.caption(
        "Internal stratified event-level validation only; "
        "not chronological, public-test, or operational validation."
    )
    st.subheader("Validation Configuration")

    config_1, config_2, config_3, config_4 = st.columns(4)

    config_1.metric(
        "Training Events",
        f"{metrics['training_events']:,}",
    )
    config_2.metric(
        "Validation Events",
        f"{metrics['validation_events']:,}",
    )
    config_3.metric(
        "Model Features",
        str(metrics["feature_count"]),
    )
    config_4.metric(
        "Prediction Cutoff",
        f"{metrics['prediction_cutoff_days']:.1f} days",
    )

    st.caption(
        f"Validation method: {metrics['validation_method']} | "
        f"Random state: {metrics['random_state']} | "
        f"Risk floor: {metrics['risk_floor']}"
    )

    st.divider()
    st.subheader("Stage 1 — Risk-Floor Classification")
    col_1, col_2, col_3 = st.columns(3)

    col_1.metric("Accuracy", f"{classifier['accuracy']:.2%}")
    col_2.metric("Precision", f"{classifier['precision']:.2%}")
    col_3.metric("Recall", f"{classifier['recall']:.2%}")

    col_4, col_5, col_6 = st.columns(3)

    col_4.metric("F1 Score", f"{classifier['f1']:.2%}")
    col_5.metric("ROC-AUC", f"{classifier['roc_auc']:.2%}")
    col_6.metric(
        "Average Precision",
        f"{classifier['average_precision']:.2%}",
    )

    st.caption(
        "The classifier separates events at the ESA dataset's "
        "-30 logarithmic risk floor from events above that floor."
    )

    st.subheader("Classification Confusion Matrix")

    confusion = pd.DataFrame(
        classifier["confusion_matrix"],
        index=["Actual above floor", "Actual risk floor"],
        columns=["Predicted above floor", "Predicted risk floor"],
    )

    st.dataframe(
        confusion,
        use_container_width=True,
    )

    st.write(
        "Majority-class baseline accuracy: "
        f"{classifier_baseline['accuracy']:.2%} | "
        "F1: "
        f"{classifier_baseline['f1']:.2%}"
    )

    st.divider()
    st.subheader("Stage 2 — Conditional Risk Regression")

    st.metric(
        "Regression Evaluation Events",
        f"{regression['evaluation_events']:,}",
    )

    regression_comparison = pd.DataFrame(
        [
            {
                "Method": "Two-stage model regressor",
                "MAE": regression_model["mae"],
                "RMSE": regression_model["rmse"],
            },
            {
                "Method": "Median baseline",
                "MAE": median_baseline["mae"],
                "RMSE": median_baseline["rmse"],
            },
            {
                "Method": "Latest observed risk baseline",
                "MAE": observed_baseline["mae"],
                "RMSE": observed_baseline["rmse"],
            },
        ]
    )

    st.dataframe(
        regression_comparison,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Conditional regression evaluates the risk regressor on "
        "above-floor events. These results do not represent the "
        "complete two-stage pipeline."
    )

    st.divider()
    st.subheader("Combined Two-Stage Prediction Performance")

    combined_col_1, combined_col_2 = st.columns(2)

    combined_col_1.metric(
        "Two-Stage Model MAE",
        f"{combined_model['mae']:.4f}",
    )

    combined_col_2.metric(
        "Two-Stage Model RMSE",
        f"{combined_model['rmse']:.4f}",
    )

    combined_comparison = pd.DataFrame(
        [
            {
                "Method": "Two-stage model",
                "MAE": combined_model["mae"],
                "RMSE": combined_model["rmse"],
            },
            {
                "Method": "Latest observed risk baseline",
                "MAE": combined_baseline["mae"],
                "RMSE": combined_baseline["rmse"],
            },
        ]
    )

    st.dataframe(
        combined_comparison,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        f"Combined evaluation covers {combined['validation_events']:,} "
        "validation events and includes the classification and "
        "conditional regression stages."
    )

    st.divider()
    st.subheader("Ablation Study Availability")

    st.info(
        "A formal feature-ablation experiment has not yet been conducted "
        "for this ESA model. No feature-removal performance results are "
        "reported here. Baseline comparisons above are included as "
        "reference benchmarks, not as feature ablations."
    )

    st.divider()
    st.subheader("Scientific Validation Notes")

    st.markdown(
        """
        - Training and validation were separated at event level.
        - The holdout was stratified and randomly sampled.
        - Validation was not chronological.
        - The private evaluation labels were not used.
        - The -30 value represents the dataset's risk floor, not zero
          collision probability.
        - The reported metrics do not establish operational readiness.
        - The model is intended for historical offline research and
          advisory decision support.
        """
    )

    st.caption(
        f"Model: {metadata.get('model_name', 'ESA Two-Stage Risk Model')} | "
        f"Dataset: {metadata.get('dataset', metrics['dataset'])}"
    )
def render_esa_audit(workspace):
    inputs = workspace["inputs"]
    predictions = workspace["predictions"]
    features = workspace["features"]
    metadata = workspace["metadata"]

    st.title("Audit & Data Quality")
    st.caption("ESA dataset provenance, integrity and inference audit")

    st.subheader("Dataset Provenance")

    provenance_1, provenance_2 = st.columns(2)

    provenance_1.metric("Dataset", metadata["dataset"])
    provenance_2.metric("License", "CC BY 4.0")

    st.markdown(
        f"**Source:** [{metadata['dataset_source']}]"
        f"({metadata['dataset_source']})"
    )

    st.write(
        "**Model:**",
        metadata["model_name"],
    )

    st.write(
        "**Inference status:**",
        metadata["operational_status"],
    )

    st.write(
        "**Prediction target:**",
        metadata["target"],
    )

    st.info(
        "The dashboard uses public historical ESA CDM inputs for offline "
        "inference. Private evaluation labels are not loaded or used."
    )

    st.divider()
    st.subheader("Input Dataset Integrity")

    total_rows = len(inputs)
    total_columns = len(inputs.columns)
    total_events = inputs["event_id"].nunique()

    integrity_1, integrity_2, integrity_3 = st.columns(3)

    integrity_1.metric("Public CDM Records", f"{total_rows:,}")
    integrity_2.metric("Input Features", f"{total_columns:,}")
    integrity_3.metric("Unique Events", f"{total_events:,}")

    missing_by_column = inputs.isna().sum()

    quality_report = pd.DataFrame(
        {
            "Column": inputs.columns,
            "Data Type": inputs.dtypes.astype(str).values,
            "Missing Values": missing_by_column.values,
            "Missing Percentage": (
                missing_by_column.values / max(total_rows, 1) * 100
            ).round(2),
        }
    )

    quality_report = quality_report.sort_values(
        "Missing Values",
        ascending=False,
    )

    st.subheader("Column-Level Missingness")

    st.dataframe(
        quality_report,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download Input Quality Report",
        data=quality_report.to_csv(index=False).encode("utf-8"),
        file_name="esa_input_quality_report.csv",
        mime="text/csv",
        key="esa_quality_report_download",
    )

    st.divider()
    st.subheader("Prediction Eligibility Audit")

    cutoff = metadata["prediction_cutoff_days"]

    below_cutoff = int((inputs["time_to_tca"] < cutoff).sum())
    eligible_events = int(features.index.nunique())
    predicted_events = int(predictions["event_id"].nunique())

    eligibility_1, eligibility_2, eligibility_3 = st.columns(3)

    eligibility_1.metric(
        "Prediction Cutoff",
        f"{cutoff:.1f} days",
    )

    eligibility_2.metric(
        "Eligible Events",
        f"{eligible_events:,}",
    )

    eligibility_3.metric(
        "Predicted Events",
        f"{predicted_events:,}",
    )

    st.write(
        f"Source CDM records below the cutoff: **{below_cutoff:,}**"
    )

    st.write(
        f"Events without an eligible prediction: "
        f"**{eligible_events - predicted_events:,}**"
    )

    cutoff_valid = bool(
        (inputs["time_to_tca"] >= cutoff).all()
    )

    predictions_unique = bool(
        predictions["event_id"].is_unique
    )

    if cutoff_valid and predictions_unique and (
        eligible_events == predicted_events
    ):
        st.success(
            "Eligibility and prediction coverage checks passed."
        )
    else:
        st.error(
            "One or more eligibility or prediction coverage checks failed."
        )

    st.divider()
    st.subheader("Numerical Integrity Checks")

    numeric_columns = inputs.select_dtypes(
        include=[np.number]
    ).columns

    non_finite = {}

    for column in numeric_columns:
        values = inputs[column].to_numpy(dtype=float)
        count = int((~np.isfinite(values)).sum())

        if count:
            non_finite[column] = count

    non_finite_total = sum(non_finite.values())

    numerical_1, numerical_2 = st.columns(2)

    numerical_1.metric(
        "Numeric Columns",
        f"{len(numeric_columns):,}",
    )

    numerical_2.metric(
        "Non-Finite Numeric Values",
        f"{non_finite_total:,}",
    )

    if non_finite:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Column": column,
                        "Non-finite Values": count,
                    }
                    for column, count in non_finite.items()
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("All numeric input values are finite.")

    st.divider()
    st.subheader("Model & Validation Record")

    audit_metadata = {
        "Model Name": metadata["model_name"],
        "Dataset": metadata["dataset"],
        "Dataset License": metadata["dataset_license"],
        "Prediction Cutoff (days)": metadata["prediction_cutoff_days"],
        "Risk Floor": metadata["risk_floor"],
        "Feature Count": metadata["feature_count"],
        "Training Events": metadata["training_events"],
        "Validation Events": metadata["validation_events"],
        "Validation Method": metadata["validation_method"],
        "Random State": metadata["random_state"],
        "Private Evaluation Labels Used": metadata[
            "private_evaluation_labels_used"
        ],
        "Operational Status": metadata["operational_status"],
    }

    st.dataframe(
        pd.DataFrame(
            [
                {"Property": key, "Value": value}
                for key, value in audit_metadata.items()
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    if metadata["private_evaluation_labels_used"] is False:
        st.success(
            "Private evaluation label usage: confirmed false in model metadata."
        )

    st.caption(
        "This audit reflects the public input data and model metadata "
        "loaded by the dashboard. It is not an independent external audit "
        "or operational certification."
    )
def render_esa_geometry(workspace):
    inputs = workspace["inputs"]
    features = workspace["features"].reset_index()
    predictions = workspace["predictions"]

    st.title("Encounter Geometry & Orbital Intelligence")
    st.caption(
        "ESA encounter measurements | Relative motion | "
        "Covariance-based uncertainty evidence"
    )

    st.info(
        "This view presents historical ESA encounter geometry. "
        "It does not simulate or display propagated orbital trajectories."
    )

    selected_event = select_esa_event(
        predictions,
        "esa_geometry_event",
    )
    if selected_event is None:
        return
    event_history = inputs[
        inputs["event_id"].astype(str) == selected_event
    ].sort_values("time_to_tca", ascending=True)

    feature_match = features[
        features["event_id"].astype(str) == selected_event
    ]

    prediction = predictions[
        predictions["event_id"].astype(str) == selected_event
    ].iloc[0]

    if event_history.empty or feature_match.empty:
        st.error("Encounter geometry evidence is unavailable.")
        return

    evidence = feature_match.iloc[0]
    # NASA Blue Marble Earth with immersive space environment
    import plotly.graph_objects as go
    from PIL import Image
    from pathlib import Path

    st.subheader("Earth Reference Globe")

    texture_path = (
        Path(__file__).resolve().parents[1]
        / "assets"
        / "earth"
        / "earth_blue_marble.jpg"
    )

    earth_image = Image.open(texture_path).convert("RGB")
    earth_image = earth_image.resize(
        (180, 90), Image.Resampling.BILINEAR
    )
    rgb = np.asarray(earth_image)

    rows, cols, _ = rgb.shape
    lat = np.linspace(np.pi / 2, -np.pi / 2, rows)
    lon = np.linspace(-np.pi, np.pi, cols)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    x = np.cos(lat_grid) * np.cos(lon_grid)
    y = np.cos(lat_grid) * np.sin(lon_grid)
    z = np.sin(lat_grid)

    vertex_colors = [
        f"rgb({r},{g},{b})"
        for r, g, b in rgb.reshape(-1, 3)
    ]

    face_i, face_j, face_k = [], [], []

    for row in range(rows - 1):
        for col in range(cols - 1):
            a = row * cols + col
            b = a + cols
            face_i.extend([a, a + 1])
            face_j.extend([b, b])
            face_k.extend([a + 1, b + 1])

    globe = go.Figure()

    globe.add_trace(go.Mesh3d(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        i=face_i,
        j=face_j,
        k=face_k,
        vertexcolor=vertex_colors,
        flatshading=False,
        lighting=dict(
            ambient=0.42,
            diffuse=0.85,
            specular=0.3,
            roughness=0.8,
        ),
        lightposition=dict(x=100, y=200, z=300),
        name="NASA Blue Marble Earth",
        hoverinfo="skip",
        showscale=False,
    ))

    # Atmospheric glow
    atmosphere_lat = np.linspace(-np.pi / 2, np.pi / 2, 45)
    atmosphere_lon = np.linspace(-np.pi, np.pi, 90)
    alon, alat = np.meshgrid(atmosphere_lon, atmosphere_lat)
    atmosphere_radius = 1.045

    globe.add_trace(go.Surface(
        x=atmosphere_radius * np.cos(alat) * np.cos(alon),
        y=atmosphere_radius * np.cos(alat) * np.sin(alon),
        z=atmosphere_radius * np.sin(alat),
        surfacecolor=np.ones_like(alat),
        colorscale=[[0, "#168BFF"], [1, "#168BFF"]],
        opacity=0.09,
        showscale=False,
        hoverinfo="skip",
        name="Atmosphere",
    ))

    # Illustrative deep-space starfield
    rng = np.random.default_rng(42)
    star_directions = rng.normal(size=(450, 3))
    star_directions /= np.linalg.norm(
        star_directions, axis=1, keepdims=True
    )
    star_positions = star_directions * rng.uniform(
        2.4, 4.5, size=(450, 1)
    )

    globe.add_trace(go.Scatter3d(
        x=star_positions[:, 0],
        y=star_positions[:, 1],
        z=star_positions[:, 2],
        mode="markers",
        marker=dict(
            size=1.5,
            color="#D6E8FF",
            opacity=0.75,
        ),
        name="Space background",
        hoverinfo="skip",
    ))

    globe.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=5, b=0),
        paper_bgcolor="#020611",
        scene=dict(
            bgcolor="#020611",
            xaxis=dict(visible=False, range=[-4.5, 4.5]),
            yaxis=dict(visible=False, range=[-4.5, 4.5]),
            zaxis=dict(visible=False, range=[-4.5, 4.5]),
            aspectmode="cube",
            camera=dict(
                eye=dict(x=1.65, y=1.65, z=1.15)
            ),
        ),
    )

    st.plotly_chart(globe, use_container_width=True)

    st.caption(
        "NASA Blue Marble Earth texture with illustrative "
        "atmosphere and starfield. Reference visualization only; "
        "no event-specific position or propagated orbit is shown."
    )

    st.markdown(
        "[Earth texture source: NASA Blue Marble Next Generation]"
        "(https://assets.science.nasa.gov/content/dam/science/esd/eo/images/"
        "bmng/bmng-base/july/world.200407.3x5400x2700.jpg)"
    )

    st.subheader("Encounter Summary")

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)

    metric_1.metric(
        "Miss Distance",
        f"{evidence['miss_distance']:,.2f} m",
    )

    metric_2.metric(
        "Relative Speed",
        f"{evidence['relative_speed']:,.2f} m/s",
    )

    metric_3.metric(
        "Mahalanobis Distance",
        f"{evidence['mahalanobis_distance']:.4f}",
    )

    metric_4.metric(
        "Predicted Final Log Risk",
        f"{prediction['predicted_final_risk']:.4f}",
    )

    st.divider()
    st.subheader("Relative Position — R/T/N Components")

    position_columns = [
        "relative_position_r",
        "relative_position_t",
        "relative_position_n",
    ]

    position_values = {
        column: evidence[column]
        for column in position_columns
        if column in evidence.index
        and pd.notna(evidence[column])
    }

    if position_values:
        position_frame = pd.DataFrame(
            {
                "Component": list(position_values.keys()),
                "Value (m)": list(position_values.values()),
            }
        ).set_index("Component")

        st.bar_chart(position_frame)

        st.dataframe(
            position_frame.reset_index(),
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "R/T/N denotes radial, transverse and normal relative-position "
        "components in the encounter reference frame."
    )

    st.divider()
    st.subheader("Relative Velocity — R/T/N Components")

    velocity_columns = [
        "relative_velocity_r",
        "relative_velocity_t",
        "relative_velocity_n",
    ]

    velocity_values = {
        column: evidence[column]
        for column in velocity_columns
        if column in evidence.index
        and pd.notna(evidence[column])
    }

    if velocity_values:
        velocity_frame = pd.DataFrame(
            {
                "Component": list(velocity_values.keys()),
                "Value (m/s)": list(velocity_values.values()),
            }
        ).set_index("Component")

        st.bar_chart(velocity_frame)

        st.dataframe(
            velocity_frame.reset_index(),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.subheader("Position Covariance Evidence")

    covariance_columns = [
        "t_position_covariance_det_log10",
        "c_position_covariance_det_log10",
    ]

    covariance_values = {
        column: evidence[column]
        for column in covariance_columns
        if column in evidence.index
        and pd.notna(evidence[column])
    }

    if covariance_values:
        covariance_frame = pd.DataFrame(
            {
                "Object": list(covariance_values.keys()),
                "Log10 Covariance Determinant": list(
                    covariance_values.values()
                ),
            }
        ).set_index("Object")

        st.bar_chart(covariance_frame)

        st.dataframe(
            covariance_frame.reset_index(),
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "Covariance determinants are displayed in log10-transformed "
        "feature form, as used by the ESA model feature pipeline."
    )

    st.divider()
    st.subheader("Historical Encounter Evolution")

    history_columns = [
        column
        for column in [
            "time_to_tca",
            "miss_distance",
            "relative_speed",
            "mahalanobis_distance",
            "risk",
        ]
        if column in event_history.columns
    ]

    history_frame = event_history[history_columns].copy()

    history_frame = history_frame.set_index("time_to_tca")

    st.write("Encounter measurements across available CDM observations")

    st.dataframe(
        history_frame,
        use_container_width=True,
    )

    st.caption(
        "The historical risk column is a per-CDM input measurement. "
        "It is not the predicted final event-level risk."
    )

    st.divider()
    st.subheader("Encounter Interpretation")

    st.markdown(
        """
        - Miss distance describes the reported separation measurement.
        - Relative speed describes the reported encounter velocity.
        - R/T/N components expose relative position and velocity in
          the encounter reference frame.
        - Mahalanobis distance provides a covariance-aware separation
          measure.
        - Position covariance determinants describe uncertainty
          information associated with the two objects.
        """
    )

    st.caption(
        "Reference geometry only. Measurements do not establish "
        "collision probability or maneuver requirements. Risk outputs "
        "are historical offline research predictions."
    )

def render_esa_screen(active_screen):
    screen_renderers = {
        "Mission Overview": render_esa_mission_overview,
        "Live Risk Monitor": render_esa_risk_monitor,
        "Event Explorer": render_esa_event_explorer,
        "Event Detail": render_esa_event_detail,
        "Evidence & Explanation": render_esa_evidence,
        "Model Performance & Ablations": render_esa_model_performance,
        "Audit & Data Quality": render_esa_audit,
        "3D Orbital Intelligence": render_esa_geometry,
    }

    renderer = screen_renderers.get(active_screen)

    if renderer is None:
        st.error(f"Unknown ESA dashboard screen: {active_screen}")
        return

    try:
        workspace = load_esa_workspace()
        metadata = workspace["metadata"]

        st.markdown(
            f"**{metadata['model_name']}**"
        )
        st.caption(
            f"{metadata['dataset']} | "
            f"{metadata['operational_status']} | "
            f"{metadata['prediction_cutoff_days']:g}-day prediction cutoff"
        )
        st.caption(
            f"Target: {metadata['target']} | "
            f"Features: {metadata['feature_count']}"
        )

        renderer(workspace)
    except Exception as exc:
        st.error("Unable to load the historical ESA intelligence workspace.")
        st.exception(exc)
