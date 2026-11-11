"""
ORBIT PULSE - Inference Engine
==============================
Executes end-to-end prediction for an event snapshot at TCA-48h.
Generates structured output adhering to Appendix A Prediction Output Contract.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib

from src.snapshot_48h import EventSnapshot48h
from src.feature_engineering import extract_snapshot_features, EPSILON
from src.evidence_intelligence import get_complete_evidence_intelligence
from src.decision_engine import evaluate_decision_policy
from src.train_model import MODEL_SAVE_PATH, METADATA_SAVE_PATH
from src.data_loader import get_feature_schema_config

class OrbitPulsePredictor:
    """End-to-end inference engine for ORBIT PULSE."""
    def __init__(self, model_path: str = MODEL_SAVE_PATH, metadata_path: str = METADATA_SAVE_PATH):
        # Resolve paths from this source file, never from the shell's current directory.
        project_root = Path(__file__).resolve().parents[1]

        model_file = Path(model_path)
        metadata_file = Path(metadata_path)

        if not model_file.is_absolute():
            model_file = project_root / model_file

        if not metadata_file.is_absolute():
            metadata_file = project_root / metadata_file

        if not model_file.is_file():
            raise FileNotFoundError(
                f"Trained model artifact is missing: {model_file}. "
                "Run: python -m src.train_model from the project folder."
            )

        if not metadata_file.is_file():
            raise FileNotFoundError(
                f"Model metadata is missing: {metadata_file}. "
                "Restore the matching metadata file or retrain the model."
            )

        # Load trained model.
        try:
            self.model = joblib.load(model_file)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load model artifact '{model_file}': "
                f"{type(exc).__name__}: {exc}. "
                "Check Python/scikit-learn versions and retrain if needed."
            ) from exc

        # Load model metadata.
        with metadata_file.open("r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        metadata_features = self.metadata.get("feature_names")

        if (
            not isinstance(metadata_features, list)
            or not metadata_features
            or not all(isinstance(name, str) and name for name in metadata_features)
        ):
            raise ValueError(
                f"Invalid or missing feature_names in model metadata: {metadata_file}"
            )

        if len(metadata_features) != len(set(metadata_features)):
            raise ValueError("Model metadata contains duplicate feature names.")

        self.feature_names = metadata_features

        # Independently load the approved feature schema.
        schema_cfg = get_feature_schema_config()
        approved_features = schema_cfg.get("feature_sets", {}).get("set_d_full")

        if not isinstance(approved_features, list) or not approved_features:
            raise ValueError(
                "Approved feature set 'set_d_full' is missing or invalid "
                "in configs/feature_schema.yaml."
            )

        # Verify metadata against the approved schema, including ordering.
        if self.feature_names != approved_features:
            raise ValueError(
                "Model metadata does not match the approved set_d_full feature schema. "
                "Check feature names and ordering before running predictions."
            )

        # Verify the actual model's expected input dimensions.
        expected_count = getattr(self.model, "n_features_in_", None)

        if expected_count != len(self.feature_names):
            raise ValueError(
                f"Model expects {expected_count} features, but metadata declares "
                f"{len(self.feature_names)}."
            )

        # Verify the model's internal feature names when available.
        model_feature_names = getattr(self.model, "feature_names_in_", None)

        if model_feature_names is not None:
            if list(model_feature_names) != self.feature_names:
                raise ValueError(
                    "Model internal feature names do not match model metadata."
                )

        self.model_version = self.metadata.get(
            "model_type", "HistGradientBoostingRegressor_v1.0"
        )
        self.dataset_version = self.metadata.get(
            "dataset_version", "UNKNOWN_DATASET_VERSION"
        )
        # Resolve paths from this source file, never from the shell's current directory.
        project_root = Path(__file__).resolve().parents[1]
        model_file = Path(model_path)
        metadata_file = Path(metadata_path)
        if not model_file.is_absolute():
            model_file = project_root / model_file
        if not metadata_file.is_absolute():
            metadata_file = project_root / metadata_file
        if not model_file.is_file():
            raise FileNotFoundError(
                f"Trained model artifact is missing: {model_file}. "
                "Run: python src/train_model.py from the project folder."
            )
        if not metadata_file.is_file():
            raise FileNotFoundError(
                f"Model metadata is missing: {metadata_file}. "
                "Restore the matching metadata file or retrain the model."
            )
        try:
            self.model = joblib.load(model_file)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load model artifact '{model_file}': {type(exc).__name__}: {exc}. "
                "Check the Python/scikit-learn versions and retrain with the installed environment if needed."
            ) from exc
        with metadata_file.open("r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        if not isinstance(self.metadata.get("feature_names"), list) or not self.metadata["feature_names"]:
            raise ValueError(f"Invalid or missing feature_names in model metadata: {metadata_file}")
        self.feature_names = self.metadata["feature_names"]
        self.model_version = self.metadata.get("model_type", "HistGradientBoostingRegressor_v1.0")
        self.dataset_version = self.metadata.get("dataset_version", "SYNTHETIC_CDM_FIXTURE_v1.0")

    def predict_snapshot(self, snapshot: EventSnapshot48h) -> Dict[str, Any]:
        """
        Runs inference on an EventSnapshot48h.
        Returns a dictionary adhering to Blueprint Appendix A Data Contract.
        """
        now_utc = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")

        # If event is ineligible:
        if not snapshot.eligible or snapshot.eligible_message_count == 0:
            return {
                "event_id": snapshot.event_id,
                "eligible": False,
                "exclusion_reason": snapshot.exclusion_reason or "NO_OBSERVATIONS_AT_OR_BEFORE_TCA_48H",
                "prediction_target": "final_reported_pc",
                "predicted_value": None,
                "predicted_log_value": None,
                "risk_band": "UNCERTAIN",
                "recommendation": "FURTHER_ASSESSMENT",
                "advisory_label": "UNCERTAIN — INELIGIBLE (NO EARLY CDM HISTORY)",
                "evidence_qualifier": "INELIGIBLE EVENT — FIRST CDM OCCURRED AFTER TCA-48h",
                "data_freshness_hours": None,
                "evidence_completeness": 0.0,
                "forecast_stability": "NOT_ESTIMABLE",
                "cutoff_utc": snapshot.cutoff_utc,
                "tca_utc": snapshot.tca_utc,
                "eligible_message_count": 0,
                "excluded_message_count": snapshot.excluded_message_count,
                "dataset_version": self.dataset_version,
                "model_version": self.model_version,
                "prediction_timestamp_utc": now_utc,
                "evidence_intelligence": {
                    "freshness": {"category": "NO_DATA", "age_hours": None},
                    "completeness": {"category": "INCOMPLETE", "completeness_ratio": 0.0},
                    "stability": {"stability_status": "NOT_ESTIMABLE"}
                },
                "decision": {
                    "risk_band": "UNCERTAIN",
                    "advisory_label": "UNCERTAIN — INELIGIBLE (NO EARLY CDM)",
                    "rationale": "No observation available at or before TCA-48h cutoff. Excluded from supervised prediction."
                }
            }

        # 1. Extract features strictly from eligible messages
        feat_dict = extract_snapshot_features(snapshot)

        if feat_dict is None:
            raise ValueError(
                f"Feature extraction failed for eligible event {snapshot.event_id}."
            )

        missing_features = [
            name for name in self.feature_names if name not in feat_dict
        ]

        if missing_features:
            raise ValueError(
                f"Missing model features for event {snapshot.event_id}: "
                f"{missing_features}"
            )

        x_vec = [feat_dict[name] for name in self.feature_names]

        if not np.isfinite(np.asarray(x_vec, dtype=float)).all():
            raise ValueError(
                f"Non-finite model input detected for event {snapshot.event_id}."
            )

        x_frame = pd.DataFrame([x_vec], columns=self.feature_names)

        # 2. Predict log10(final Pc + eps)
        pred_log = float(self.model.predict(x_frame)[0])
        # Invert transform: Pc = 10^(pred_log) - eps
        pred_pc = max(0.0, min(1.0, (10 ** pred_log) - EPSILON))

        # 3. Compute Evidence Intelligence
        evidence_intel = get_complete_evidence_intelligence(snapshot, self.model, self.feature_names)

        # 4. Apply Decision Policy
        decision = evaluate_decision_policy(pred_pc, evidence_intel)

        # Map recommendation for contract
        rec_map = {
            "LOW": "ROUTINE_MONITORING",
            "UNCERTAIN": "FURTHER_ASSESSMENT",
            "HIGH": "MANOEUVRE_REVIEW"
        }
        recommendation = rec_map.get(decision["risk_band"], "FURTHER_ASSESSMENT")

        return {
            "event_id": snapshot.event_id,
            "eligible": True,
            "exclusion_reason": None,
            "prediction_target": "final_reported_pc",
            "predicted_value": float(pred_pc),
            "predicted_log_value": round(float(pred_log), 4),
            "risk_band": decision["risk_band"],
            "recommendation": recommendation,
            "advisory_label": decision["advisory_label"],
            "evidence_qualifier": decision["evidence_qualifier"],
            "data_freshness_hours": evidence_intel["freshness"].get("age_hours"),
            "evidence_completeness": evidence_intel["completeness"].get("completeness_ratio"),
            "forecast_stability": evidence_intel["stability"].get("stability_status"),
            "cutoff_utc": snapshot.cutoff_utc,
            "tca_utc": snapshot.tca_utc,
            "eligible_message_count": snapshot.eligible_message_count,
            "excluded_message_count": snapshot.excluded_message_count,
            "dataset_version": self.dataset_version,
            "model_version": self.model_version,
            "prediction_timestamp_utc": now_utc,
            "evidence_intelligence": evidence_intel,
            "decision": decision
        }
