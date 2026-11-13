from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.esa_features import build_esa_event_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ESA_MODEL_PATH = (
    PROJECT_ROOT / "models" / "esa" / "esa_two_stage_model.joblib"
)


class ESATwoStagePredictor:
    """
    Inference engine for the ORBIT PULSE ESA two-stage risk model.

    This predictor is independent of the original synthetic CDM
    prediction pipeline.

    Predictions represent final event-level base-10 logarithmic
    risk, using only CDMs available at least two days before TCA.
    """

    def __init__(self, model_path=None):
        self.model_path = (
            Path(model_path)
            if model_path is not None
            else DEFAULT_ESA_MODEL_PATH
        )

        if not self.model_path.is_absolute():
            self.model_path = PROJECT_ROOT / self.model_path

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"ESA model artifact not found: {self.model_path}"
            )

        try:
            self.bundle = joblib.load(self.model_path)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load ESA model: {type(exc).__name__}: {exc}"
            ) from exc

        required_keys = {
            "classifier",
            "regressor",
            "feature_columns",
            "numeric_columns",
            "categorical_columns",
            "prediction_cutoff_days",
            "risk_floor",
        }

        missing_keys = required_keys - set(self.bundle)

        if missing_keys:
            raise ValueError(
                f"ESA model bundle is missing components: "
                f"{sorted(missing_keys)}"
            )

        self.classifier = self.bundle["classifier"]
        self.regressor = self.bundle["regressor"]
        self.feature_columns = list(self.bundle["feature_columns"])
        self.prediction_cutoff_days = float(
            self.bundle["prediction_cutoff_days"]
        )
        self.risk_floor = float(self.bundle["risk_floor"])

        if not self.feature_columns:
            raise ValueError("ESA model has no feature columns.")

        if len(self.feature_columns) != len(set(self.feature_columns)):
            raise ValueError("ESA model contains duplicate feature columns.")

        if not np.isfinite(self.prediction_cutoff_days):
            raise ValueError("Invalid ESA prediction cutoff.")

        if self.prediction_cutoff_days < 0:
            raise ValueError("ESA prediction cutoff cannot be negative.")

        if not np.isfinite(self.risk_floor):
            raise ValueError("Invalid ESA risk floor.")

    def predict(self, event_history: pd.DataFrame) -> pd.DataFrame:
        """
        Generate one prediction per eligible event.

        Parameters
        ----------
        event_history:
            ESA CDM history containing event_id, time_to_tca,
            and the required feature and risk columns.

        Returns
        -------
        pandas.DataFrame:
            Event-level predicted final log-risk and classification.
        """

        if not isinstance(event_history, pd.DataFrame):
            raise TypeError("event_history must be a pandas DataFrame.")

        features = build_esa_event_features(
            event_history,
            prediction_cutoff_days=self.prediction_cutoff_days,
        )

        missing_columns = (
            set(self.feature_columns) - set(features.columns)
        )

        extra_columns = (
            set(features.columns) - set(self.feature_columns)
        )

        if missing_columns or extra_columns:
            raise ValueError(
                "ESA feature schema mismatch. "
                f"Missing: {sorted(missing_columns)}. "
                f"Unexpected: {sorted(extra_columns)}."
            )

        features = features.reindex(
            columns=self.feature_columns
        )

        floor_predictions = np.asarray(
            self.classifier.predict(features)
        )

        if len(floor_predictions) != len(features):
            raise ValueError("ESA classifier returned an invalid prediction count.")

        if not all(
            isinstance(value, (bool, np.bool_))
            for value in floor_predictions
        ):
            raise ValueError("ESA classifier must return boolean risk-floor labels.")

        floor_predictions = floor_predictions.astype(bool)

        predicted_risk = np.full(
            len(features),
            self.risk_floor,
            dtype=float,
        )

        above_floor_mask = ~floor_predictions

        if above_floor_mask.any():
            regression_features = features.loc[above_floor_mask]

            regression_predictions = np.asarray(
                self.regressor.predict(regression_features),
                dtype=float,
            )

            if not np.isfinite(regression_predictions).all():
                raise ValueError(
                    "ESA regressor returned non-finite predictions."
                )

            predicted_risk[above_floor_mask] = regression_predictions

        if not np.isfinite(predicted_risk).all():
            raise ValueError("ESA predictions contain non-finite values.")

        results = pd.DataFrame({
            "event_id": features.index.to_numpy(),
            "predicted_final_risk": predicted_risk,
            "predicted_risk_floor": floor_predictions,
            "selected_time_to_tca_days": (
                features["time_to_tca"].to_numpy(dtype=float)
            ),
            "prediction_cutoff_days": self.prediction_cutoff_days,
            "prediction_target": "final_event_base10_log_risk",
            "model_name": "ORBIT PULSE ESA Two-Stage Risk Model",
            "operational_status": "Historical ESA Data Model - Offline Inference",
        })

        if results["event_id"].duplicated().any():
            raise ValueError("Duplicate event predictions detected.")

        return results
