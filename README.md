# ORBIT PULSE
### Satellite Conjunction Risk Intelligence & Decision Support Prototype (SW-5)

> [!IMPORTANT]
> **OPERATIONAL & SCIENTIFIC SAFETY DISCLAIMER**
> 1. **Advisory Prototype**: ORBIT PULSE is an academic research and decision-support prototype. Qualified satellite operators make all final manoeuvre decisions. Never issue autonomous burn commands.
> 2. **Target Distinction**: A final reported Conjunction Data Message (CDM) $P_c$ is an eventual dataset-reported risk estimate, **not** proof that a physical collision occurred.
> 3. **Uncertainty Distinction**: Model sensitivity to eligible historical inputs reflects algorithmic stability under data perturbations, **not** physical orbit covariance or real-world uncertainty bounds.
> 4. **Data Provenance**: In the absence of an approved real-time operational CDM feed, this prototype uses a clearly labeled synthetic fixture for UI and pipeline verification. **Never report synthetic predictive performance as real validation.**

---

## 1. Executive Direction & System Overview

Space traffic management is not merely about calculating close approaches; it is about **making critical decisions from incomplete, noisy, and rapidly changing information**.

**ORBIT PULSE** is a time-aware satellite conjunction risk intelligence platform that:
- Reconstructs conjunction events from successive CDM histories without mixing events.
- Enforces an auditable, leakage-safe snapshot from observations available **strictly at or before TCA − 48 hours**.
- Forecasts the dataset-defined final reported collision probability ($P_c$) and benchmarks performance against a simple last-observation baseline.
- Houses a signature **Evidence Intelligence Layer** that separately reveals **Data Freshness**, **Evidence Completeness**, and **Forecast Stability** (never collapsed into an arbitrary "confidence percentage").
- Connects predictions to a transparent, advisory review policy with human-in-the-loop escalation rules.

---

## 2. Core Architecture & Data Contract

```
ORBIT-PULSE/
├── data/
│   ├── raw/             # Raw CDM files (synthetic fixture clearly labeled)
│   ├── processed/       # Cleaned, validated snapshots
│   └── demo/            # Demo replay streams
├── models/
│   ├── risk_model.joblib        # Persisted tabular risk model
│   └── model_metadata.json      # Provenance, features, and split ranges
├── notebooks/
│   └── exploration.ipynb        # Step-by-step pipeline walkthrough
├── src/
│   ├── data_loader.py           # Ingestion and configuration parsing
│   ├── validation.py            # Schema, range, timestamp, duplicate checks
│   ├── event_builder.py         # Chronological event grouping
│   ├── snapshot_48h.py          # Strict TCA-48h cutoff logic
│   ├── feature_engineering.py   # Leakage-safe feature extraction
│   ├── train_model.py           # Chronological event-separated training
│   ├── evaluate_model.py        # Metrics and mandatory ablations
│   ├── predict.py               # Appendix A prediction output contract
│   ├── evidence_intelligence.py # Freshness, completeness, stability
│   ├── decision_engine.py       # Advisory decision policy
│   └── audit.py                 # Append-only audit trail logging
├── dashboard/
│   ├── app.py                   # Streamlit Mission-Control Console
│   ├── charts.py                # Plotly visualizations
│   └── theme.py                 # Dark aerospace styling
├── tests/                       # Automated pytest suite (20 tests)
│   ├── test_cutoff.py           # Boundary & exclusion tests
│   ├── test_event_grouping.py   # Grouping and duplicate tests
│   ├── test_no_leakage.py       # Anti-leakage & split integrity tests
│   ├── test_timestamps.py       # Timestamp & anomaly tests
│   └── test_decision_policy.py  # Advisory wording & safety tests
├── configs/
│   ├── feature_schema.yaml      # Feature contract & ablation definitions
│   └── thresholds.yaml          # Operational & evidence thresholds
├── reports/
│   ├── data_quality.md          # Comprehensive evaluation report
│   └── metrics.json             # Structured performance metrics
├── requirements.txt
└── README.md
```

### The TCA − 48h Cutoff Rule
- **Boundary**: `eligible message_time <= TCA - 48 hours`.
- Messages occurring exactly on the 48-hour boundary are admitted.
- Any message arriving after `TCA - 48h` is strictly excluded from model feature extraction.
- **Ineligible Events**: If an event has no observations before `TCA - 48h`, it is marked `eligible = False` and excluded from model scoring. A later message is **never substituted**.
- **Label Isolation**: Final reported $P_c$ is retained solely as an evaluation target and never enters the feature space.

---

## 3. Evidence Intelligence Layer

Rather than presenting an uninterpretable single "confidence percentage," ORBIT PULSE computes three independent operational dimensions:

| Dimension | Definition | Categorization |
|---|---|---|
| **Data Freshness** | Latency between cutoff epoch and the latest admitted CDM | **FRESH** ($\le 12$h), **AGEING** ($12-24$h), **STALE** ($> 24$h) |
| **Evidence Completeness** | Ratio of documented schema fields present across history | **HIGH** ($\ge 90\%$), **MODERATE** ($70-90\%$), **LIMITED** ($< 70\%$) |
| **Forecast Stability** | Algorithmic sensitivity under leave-one-out CDM message perturbations | **STABLE** ($\text{shift} \le 1.0 \text{ dex}$), **SENSITIVE** ($> 1.0 \text{ dex}$), **NOT_ESTIMABLE** ($< 2$ msgs) |

---

## 4. Transparent Advisory Decision Support

Predictions and evidence are routed through a decoupled advisory policy:

- **`HIGH — ESCALATE FOR MANOEUVRE REVIEW`**:
  Triggered when forecasted $P_c \ge 10^{-4}$ (or competition-defined operational review threshold). Qualified operators prepare scenario options and assess tracking tasking.
- **`UNCERTAIN — FURTHER ASSESSMENT`**:
  Triggered when forecasted $P_c < 10^{-4}$ **but** evidence is `STALE`, `LIMITED`, or `SENSITIVE`. Prevents false sense of security; event is held for urgent re-observation.
- **`LOW — ROUTINE MONITORING`**:
  Triggered when forecasted $P_c < 10^{-4}$ **and** evidence is verified fresh and complete.

---

## 5. Quickstart & Verification Guide

### Prerequisites
- Python 3.10–3.12 recommended for the most predictable scientific Python package compatibility.

### 1. Install Dependencies
```powershell
python -m pip install -r requirements.txt
```

### 2. Run Automated Test Suite
Run the 20 automated unit and integration tests covering cutoff boundaries, leakage prevention, timestamp parsing, and safety wording:
```powershell
python -m pytest tests/ -v
```

### 3. Train & Evaluate Models (with Mandatory Ablations)
```powershell
# Chronological event-separated training
python src/train_model.py

# Model evaluation, mandatory ablations A-D, and report generation
python src/evaluate_model.py
```
This updates `reports/metrics.json` and `reports/data_quality.md`.

### 4. Launch Streamlit Mission-Control Console
```powershell
python -m streamlit run dashboard/app.py
```
Access the console in your browser at `http://localhost:8501`.

### Model loading troubleshooting
- The application now resolves model, metadata, data, configuration, and report paths relative to the project directory; launching it from a different current working directory should not cause missing-file errors.
- If the sidebar reports **MODEL OFFLINE**, open **Model loading diagnostics** and read the exact exception rather than assuming the model file is absent.
- Confirm `models/risk_model.joblib` and `models/model_metadata.json` are both present.
- If the diagnostic identifies a scikit-learn/joblib compatibility problem, activate the project's virtual environment, reinstall `requirements.txt`, then retrain and regenerate the metrics using the commands in Section 5.
- Use **Retry model loading** after correcting the issue; do not silently treat unavailable predictions as zero risk.

---

## 6. Live, Replay & Manual Modes

- **REPLAY Mode**: Replays chronological CDM messages in timestamp order. Use the sidebar controls (`Step Forward ⏩`, `Reset Stream 🔄`) to simulate incoming tracking updates.
- **LIVE FEED (SIMULATED PLACEHOLDER)**: Displays the bundled fixture with an explicit simulated-feed label. It does not connect to an operational CDM source or claim real-time ingestion.
- **MANUAL / BATCH Mode**: Upload custom CSV CDM files for schema validation, cutoff snapshot reconstruction, and decision scoring.
- **Event comparison & exports**: Compare up to four filtered events and download event registers, event histories, evaluation metrics, validation audits, and audit trails.

---

## 7. 3–5 Minute Judge Demo Script

1. **Introduction (1 min)**:
   - Introduce ORBIT PULSE's problem definition: Conjunction risk forecasting at the strict `TCA - 48h` operational decision boundary.
   - Point out the persistent **Operational Mode Badge** and the honest **Data Provenance notice**.
2. **Mission Overview & Live Monitor (1 min)**:
   - Walk through the KPI cards (Ingested CDMs, Distinct Events, Eligible vs Ineligible counts).
   - Switch to **Live Risk Monitor** and click `Step Forward ⏩` in Replay Mode to demonstrate dynamic feed arrival and updating event risk timelines.
3. **Event Detail & Cutoff Marker (1 min)**:
   - Navigate to **Event Detail**. Show the interactive Plotly chart with the dashed red vertical line at `TCA - 48h`.
   - Point out the distinction: cyan points (admitted pre-cutoff CDMs) vs muted open points (excluded post-cutoff CDMs). Show the forecasted marker alongside ground truth.
4. **Evidence Intelligence & Decision Policy (1 min)**:
   - Open **Evidence & Explanation**. Show the 3 independent indicators: Freshness, Completeness, and Stability.
   - Review the **Leave-One-Message-Out Perturbation Plot** and explain that sensitivity measures algorithmic stability, not physical orbit uncertainty.
   - Explain why a low numerical risk with stale data maps to `UNCERTAIN — FURTHER ASSESSMENT` rather than an all-clear.
5. **Model Evaluation & Disclaimer (30s)**:
   - Show **Model Performance & Ablations**: Benchmark comparison against the Last-Eligible Pc baseline, the 4 mandatory ablation tiers (A, B, C, D), and the confusion matrix.
   - Conclude with the operational disclaimer: advisory decision support for human operators.
