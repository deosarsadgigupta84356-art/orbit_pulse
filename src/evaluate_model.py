"""
ORBIT PULSE - Model Evaluation & Ablation Suite
===============================================
Evaluates models strictly on chronological held-out test events.
Computes:
- Regression metrics (MAE, RMSE, Median AE in log10 space)
- Classification metrics (Recall, Precision, FN, FP, PR-AUC for high-risk Pc >= 1e-4)
- Mandatory Ablations (A, B, C, D)
- Generates reports/metrics.json and reports/data_quality.md
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    precision_score,
    recall_score,
    confusion_matrix,
    precision_recall_curve,
    auc
)
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from src.data_loader import load_raw_cdm_data, get_feature_schema_config
from src.validation import validate_cdm_records
from src.event_builder import build_event_histories
from src.snapshot_48h import build_all_snapshots
from src.feature_engineering import build_dataset_feature_matrix
from src.train_model import chronological_event_split, MODEL_SAVE_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_REPORT_PATH = str(PROJECT_ROOT / "reports" / "metrics.json")
DATA_QUALITY_REPORT_PATH = str(PROJECT_ROOT / "reports" / "data_quality.md")

def evaluate_predictions(y_true_log: np.ndarray, y_pred_log: np.ndarray, threshold_log: float = -4.0) -> Dict[str, Any]:
    """Computes comprehensive regression and operational classification metrics."""
    mae = float(mean_absolute_error(y_true_log, y_pred_log))
    rmse = float(root_mean_squared_error(y_true_log, y_pred_log))
    median_ae = float(np.median(np.abs(y_true_log - y_pred_log)))

    # Binary dangerous classification at threshold (log10(1e-4) = -4.0)
    y_true_binary = (y_true_log >= threshold_log).astype(int)
    y_pred_binary = (y_pred_log >= threshold_log).astype(int)

    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    recall = float(recall_score(y_true_binary, y_pred_binary, zero_division=0))
    precision = float(precision_score(y_true_binary, y_pred_binary, zero_division=0))

    # PR curve & AUC
    if len(np.unique(y_true_binary)) > 1:
        precisions, recalls, _ = precision_recall_curve(y_true_binary, y_pred_log)
        pr_auc = float(auc(recalls, precisions))
    else:
        pr_auc = 1.0 if tp > 0 else 0.0

    return {
        "mae_log10": round(mae, 4),
        "rmse_log10": round(rmse, 4),
        "median_ae_log10": round(median_ae, 4),
        "classification": {
            "threshold_log10": threshold_log,
            "threshold_pc": 10 ** threshold_log,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "pr_auc": round(pr_auc, 4)
        }
    }

def run_mandatory_ablations(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    schema_cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Executes mandatory ablation experiments A, B, C, D:
    A: Last eligible Pc only
    B: Pc history summaries
    C: History + Geometry + Uncertainty
    D: Full features (History + Geometry + Uncertainty + Evidence)
    """
    feature_sets = {
        "A_baseline_last_pc": schema_cfg["feature_sets"]["set_a_baseline"],
        "B_pc_history": schema_cfg["feature_sets"]["set_b_history"],
        "C_history_geometry_cov": schema_cfg["feature_sets"]["set_c_context"],
        "D_full_evidence": schema_cfg["feature_sets"]["set_d_full"]
    }

    ablation_results = {}

    y_train = train_df["target_final_log_pc"].values
    y_test = test_df["target_final_log_pc"].values

    for exp_name, cols in feature_sets.items():
        if exp_name == "A_baseline_last_pc":
            # Simple baseline: Last eligible Pc is the prediction itself
            y_pred_test = test_df["last_eligible_pc_log"].values
        else:
            X_train = train_df[cols].fillna(0.0)
            X_test = test_df[cols].fillna(0.0)

            # Train candidate model
            model = HistGradientBoostingRegressor(
                max_iter=80,
                max_leaf_nodes=15,
                min_samples_leaf=2,
                learning_rate=0.08,
                random_state=42
            )
            model.fit(X_train, y_train)
            y_pred_test = model.predict(X_test)

        metrics = evaluate_predictions(y_test, y_pred_test)
        ablation_results[exp_name] = {
            "feature_columns": cols,
            "feature_count": len(cols),
            "metrics": metrics
        }

    return ablation_results

def run_complete_evaluation() -> Dict[str, Any]:
    """Runs complete chronological evaluation and writes reports."""
    schema_cfg = get_feature_schema_config()
    feature_cols = schema_cfg["feature_sets"]["set_d_full"]

    raw_df = load_raw_cdm_data()
    valid_df, quarantine_df, val_audit = validate_cdm_records(raw_df)
    events = build_event_histories(valid_df)
    snapshots = build_all_snapshots(events)
    df_feat = build_dataset_feature_matrix(snapshots)

    train_df, val_df, test_df = chronological_event_split(df_feat, 0.60, 0.20)

    # Evaluate Baseline (Last eligible Pc)
    y_test = test_df["target_final_log_pc"].values
    y_baseline_pred = test_df["last_eligible_pc_log"].values
    baseline_eval = evaluate_predictions(y_test, y_baseline_pred)

    # Load trained ML model
    if not os.path.exists(MODEL_SAVE_PATH):
        from src.train_model import train_and_save_models
        train_and_save_models()

    model = joblib.load(MODEL_SAVE_PATH)
    X_test = test_df[feature_cols].fillna(0.0)
    y_ml_pred = model.predict(X_test)
    ml_eval = evaluate_predictions(y_test, y_ml_pred)

    # Retain held-out test curve points and absolute errors for dashboard inspection.
    y_test_binary = (y_test >= -4.0).astype(int)
    baseline_precision, baseline_recall, baseline_thresholds = precision_recall_curve(
        y_test_binary, y_baseline_pred
    )
    ml_precision, ml_recall, ml_thresholds = precision_recall_curve(
        y_test_binary, y_ml_pred
    )
    baseline_pr_curve = {
        "precision": baseline_precision.tolist(),
        "recall": baseline_recall.tolist(),
        "thresholds": baseline_thresholds.tolist()
    }
    ml_pr_curve = {
        "precision": ml_precision.tolist(),
        "recall": ml_recall.tolist(),
        "thresholds": ml_thresholds.tolist()
    }

    # Relative improvement
    mae_diff = baseline_eval["mae_log10"] - ml_eval["mae_log10"]
    rel_improvement_pct = (mae_diff / baseline_eval["mae_log10"]) * 100.0 if baseline_eval["mae_log10"] > 0 else 0.0

    # Run ablations
    ablations = run_mandatory_ablations(train_df, val_df, test_df, schema_cfg)

    evaluation_summary = {
        "dataset_metadata": {
            "is_synthetic_fixture": raw_df.attrs.get("is_synthetic_fixture", True),
            "dataset_version": raw_df.attrs.get("dataset_version", "UNKNOWN"),
            "total_raw_rows": len(raw_df),
            "valid_rows": len(valid_df),
            "quarantined_rows": len(quarantine_df),
            "total_events": len(events),
            "eligible_events": len(df_feat),
            "ineligible_events": len(events) - len(df_feat)
        },
        "chronological_split": {
            "train_events": len(train_df),
            "val_events": len(val_df),
            "test_events": len(test_df)
        },
        "threshold_policy": {
            "pc_threshold": 1e-4,
            "selection_status": "FIXED_CONFIGURED_NOT_VALIDATION_SELECTED",
            "reason": "Synthetic fixture is for pipeline/UI verification only; threshold tuning is deferred until an approved, sufficiently representative validation dataset is available."
        },
        "baseline_model": {
            "name": "Last Eligible Pc Baseline",
            "metrics": baseline_eval,
            "pr_curve": baseline_pr_curve,
            "absolute_error_log10": np.abs(y_test - y_baseline_pred).tolist()
        },
        "trained_ml_model": {
            "name": "HistGradientBoostingRegressor (Ablation D)",
            "metrics": ml_eval,
            "pr_curve": ml_pr_curve,
            "absolute_error_log10": np.abs(y_test - y_ml_pred).tolist(),
            "relative_mae_improvement_pct": round(rel_improvement_pct, 2)
        },
        "mandatory_ablations": ablations,
        "validation_audit": val_audit
    }

    # Write metrics.json
    os.makedirs(os.path.dirname(METRICS_REPORT_PATH), exist_ok=True)
    with open(METRICS_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(evaluation_summary, f, indent=2)

    # Write data_quality.md
    write_data_quality_report(evaluation_summary)

    print(f"Evaluation finished. Metrics saved to {METRICS_REPORT_PATH} and {DATA_QUALITY_REPORT_PATH}")
    return evaluation_summary

def write_data_quality_report(summary: Dict[str, Any]):
    """Generates the formal Markdown data quality & evaluation audit report."""
    ds = summary["dataset_metadata"]
    audit = summary["validation_audit"]
    base_m = summary["baseline_model"]["metrics"]
    ml_m = summary["trained_ml_model"]["metrics"]
    abl = summary["mandatory_ablations"]

    provenance_banner = (
        "> [!IMPORTANT]\n"
        "> **SYNTHETIC TEST FIXTURE AUDIT NOTICE**\n"
        "> This report is compiled from a synthetic fixture for automated pipeline and UI verification.\n"
        "> As required by Blueprint Section 1 & 16, synthetic predictive metrics must NEVER be reported as real-world operational validation."
        if ds["is_synthetic_fixture"] else
        "> [!NOTE]\n"
        "> **APPROVED HISTORICAL CDM AUDIT REPORT**\n"
        "> Evaluated on approved historical conjunction event sequences."
    )

    md_content = f"""# ORBIT PULSE - Data Quality & Model Evaluation Report

{provenance_banner}

## 1. Dataset & Ingestion Summary
- **Dataset Version**: `{ds['dataset_version']}`
- **Total Raw Records Ingested**: {ds['total_raw_rows']}
- **Valid CDM Records**: {ds['valid_rows']}
- **Quarantined Records**: {ds['quarantined_rows']}
- **Duplicates Filtered**: {audit['duplicate_records_count']}
- **Total Distinct Events**: {ds['total_events']}
- **Eligible Events (<= TCA-48h)**: {ds['eligible_events']}
- **Ineligible Events**: {ds['ineligible_events']} (No CDM before TCA-48h cutoff)

## 2. Chronological Split (Event-Separated)
No conjunction event appears in more than one split partition. Events are partitioned chronologically by Time of Closest Approach (TCA):
- **Training Events**: {summary['chronological_split']['train_events']}
- **Validation Events**: {summary['chronological_split']['val_events']}
- **Held-out Test Events**: {summary['chronological_split']['test_events']}

## 3. Classification Threshold Policy
- **Configured review threshold**: Pc >= 1e-4.
- **Selection status**: Not validation-selected in this prototype run.
- **Reason**: The bundled data is a synthetic test fixture and is not suitable for claiming an operationally validated threshold. Revisit threshold selection using an approved, representative validation dataset and documented missed-danger vs unnecessary-review costs.

## 4. Benchmark Comparison on Held-Out Test Set

| Metric | Last Eligible Pc Baseline | Trained ML Model (Ablation D) |
|---|---|---|
| **MAE (log10 space)** | {base_m['mae_log10']} | {ml_m['mae_log10']} |
| **RMSE (log10 space)** | {base_m['rmse_log10']} | {ml_m['rmse_log10']} |
| **Median Absolute Error** | {base_m['median_ae_log10']} | {ml_m['median_ae_log10']} |
| **Operational Recall (Pc >= 1e-4)** | {base_m['classification']['recall']} | {ml_m['classification']['recall']} |
| **Operational Precision** | {base_m['classification']['precision']} | {ml_m['classification']['precision']} |
| **False Negatives (Missed Danger)** | {base_m['classification']['fn']} | {ml_m['classification']['fn']} |
| **False Positives (Unnecessary Escalation)** | {base_m['classification']['fp']} | {ml_m['classification']['fp']} |
| **PR-AUC** | {base_m['classification']['pr_auc']} | {ml_m['classification']['pr_auc']} |

## 5. Interpretation & Limitations
- The ML model's MAE is {ml_m['mae_log10']:.4f}, compared with {base_m['mae_log10']:.4f} for the last-Pc baseline. The ML model therefore does **not** outperform the baseline on MAE in this test split.
- The ML model has RMSE {ml_m['rmse_log10']:.4f} versus {base_m['rmse_log10']:.4f}, while its median absolute error is {ml_m['median_ae_log10']:.4f} versus {base_m['median_ae_log10']:.4f}.
- The held-out test contains only {summary['chronological_split']['test_events']} events. Reported classification metrics are highly sample-sensitive and must not be generalised.
- The full-feature ablation uses the same model hyperparameters as the persisted model so its result is directly comparable.

## 6. Mandatory Ablation Study

| Experiment | Feature Set | Features | MAE (log10) | RMSE (log10) | Operational Recall |
|---|---|---|---|---|---|
| **A (Baseline)** | Last Eligible Pc Only | {abl['A_baseline_last_pc']['feature_count']} | {abl['A_baseline_last_pc']['metrics']['mae_log10']} | {abl['A_baseline_last_pc']['metrics']['rmse_log10']} | {abl['A_baseline_last_pc']['metrics']['classification']['recall']} |
| **B** | Pc History Summaries | {abl['B_pc_history']['feature_count']} | {abl['B_pc_history']['metrics']['mae_log10']} | {abl['B_pc_history']['metrics']['rmse_log10']} | {abl['B_pc_history']['metrics']['classification']['recall']} |
| **C** | History + Geometry/Covariance | {abl['C_history_geometry_cov']['feature_count']} | {abl['C_history_geometry_cov']['metrics']['mae_log10']} | {abl['C_history_geometry_cov']['metrics']['rmse_log10']} | {abl['C_history_geometry_cov']['metrics']['classification']['recall']} |
| **D (Full)** | History + Geometry + Evidence | {abl['D_full_evidence']['feature_count']} | {abl['D_full_evidence']['metrics']['mae_log10']} | {abl['D_full_evidence']['metrics']['rmse_log10']} | {abl['D_full_evidence']['metrics']['classification']['recall']} |

## 7. Safety & Operational Guardrails
- **Advisory Prototype**: All outputs are advisory decision-support indicators. Qualified spacecraft operators make real flight decisions.
- **Strict Leakage Prevention**: Features are extracted exclusively from records with `message_time <= TCA - 48h`.
- **Target Distinction**: Final reported Pc is a dataset-defined risk estimate, not an actual collision event.
"""

    with open(DATA_QUALITY_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(md_content)

if __name__ == "__main__":
    run_complete_evaluation()
