# ORBIT PULSE - Data Quality & Model Evaluation Report

> [!IMPORTANT]
> **SYNTHETIC TEST FIXTURE AUDIT NOTICE**
> This report is compiled from a synthetic fixture for automated pipeline and UI verification.
> As required by Blueprint Section 1 & 16, synthetic predictive metrics must NEVER be reported as real-world operational validation.

## 1. Dataset & Ingestion Summary
- **Dataset Version**: `SYNTHETIC_CDM_FIXTURE_v1.0`
- **Total Raw Records Ingested**: 223
- **Valid CDM Records**: 222
- **Quarantined Records**: 0
- **Duplicates Filtered**: 1
- **Total Distinct Events**: 40
- **Eligible Events (<= TCA-48h)**: 36
- **Ineligible Events**: 4 (No CDM before TCA-48h cutoff)

## 2. Chronological Split (Event-Separated)
No conjunction event appears in more than one split partition. Events are partitioned chronologically by Time of Closest Approach (TCA):
- **Training Events**: 21
- **Validation Events**: 7
- **Held-out Test Events**: 8

## 3. Classification Threshold Policy
- **Configured review threshold**: Pc >= 1e-4.
- **Selection status**: Not validation-selected in this prototype run.
- **Reason**: The bundled data is a synthetic test fixture and is not suitable for claiming an operationally validated threshold. Revisit threshold selection using an approved, representative validation dataset and documented missed-danger vs unnecessary-review costs.

## 4. Benchmark Comparison on Held-Out Test Set

| Metric | Last Eligible Pc Baseline | Trained ML Model (Ablation D) |
|---|---|---|
| **MAE (log10 space)** | 0.077 | 0.096 |
| **RMSE (log10 space)** | 0.1149 | 0.1115 |
| **Median Absolute Error** | 0.0566 | 0.1098 |
| **Operational Recall (Pc >= 1e-4)** | 1.0 | 1.0 |
| **Operational Precision** | 1.0 | 1.0 |
| **False Negatives (Missed Danger)** | 0 | 0 |
| **False Positives (Unnecessary Escalation)** | 0 | 0 |
| **PR-AUC** | 1.0 | 1.0 |

## 5. Interpretation & Limitations
- The ML model's MAE is 0.0960, compared with 0.0770 for the last-Pc baseline. The ML model therefore does **not** outperform the baseline on MAE in this test split.
- The ML model has RMSE 0.1115 versus 0.1149, while its median absolute error is 0.1098 versus 0.0566.
- The held-out test contains only 8 events. Reported classification metrics are highly sample-sensitive and must not be generalised.
- The full-feature ablation uses the same model hyperparameters as the persisted model so its result is directly comparable.

## 6. Mandatory Ablation Study

| Experiment | Feature Set | Features | MAE (log10) | RMSE (log10) | Operational Recall |
|---|---|---|---|---|---|
| **A (Baseline)** | Last Eligible Pc Only | 1 | 0.077 | 0.1149 | 1.0 |
| **B** | Pc History Summaries | 7 | 0.0964 | 0.1246 | 1.0 |
| **C** | History + Geometry/Covariance | 13 | 0.096 | 0.1115 | 1.0 |
| **D (Full)** | History + Geometry + Evidence | 16 | 0.096 | 0.1115 | 1.0 |

## 7. Safety & Operational Guardrails
- **Advisory Prototype**: All outputs are advisory decision-support indicators. Qualified spacecraft operators make real flight decisions.
- **Strict Leakage Prevention**: Features are extracted exclusively from records with `message_time <= TCA - 48h`.
- **Target Distinction**: Final reported Pc is a dataset-defined risk estimate, not an actual collision event.
