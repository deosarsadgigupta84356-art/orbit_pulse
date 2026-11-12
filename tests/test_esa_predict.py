import numpy as np
import pandas as pd
import pytest

from src.esa_data import load_esa_test_data
from src.esa_features import build_esa_event_features
from src.esa_predict import ESATwoStagePredictor


class FakeClassifier:
    def predict(self, features):
        # First event is risk-floor; remaining events are above-floor.
        return np.array(
            [True] + [False] * (len(features) - 1),
            dtype=bool,
        )


class FakeRegressor:
    def predict(self, features):
        return np.full(len(features), -10.0, dtype=float)


@pytest.fixture
def predictor_and_history():
    public_data = load_esa_test_data()

    # Use complete official public-test feature schema.
    selected_ids = public_data["event_id"].drop_duplicates().iloc[:3]
    history = public_data[
        public_data["event_id"].isin(selected_ids)
    ].copy()

    expected_features = build_esa_event_features(
        history,
        prediction_cutoff_days=2.0,
    )

    bundle = {
        "classifier": FakeClassifier(),
        "regressor": FakeRegressor(),
        "feature_columns": list(expected_features.columns),
        "numeric_columns": [],
        "categorical_columns": [],
        "prediction_cutoff_days": 2.0,
        "risk_floor": -30.0,
    }

    predictor = ESATwoStagePredictor.__new__(
        ESATwoStagePredictor
    )
    predictor.model_path = None
    predictor.bundle = bundle
    predictor.classifier = bundle["classifier"]
    predictor.regressor = bundle["regressor"]
    predictor.feature_columns = bundle["feature_columns"]
    predictor.prediction_cutoff_days = 2.0
    predictor.risk_floor = -30.0

    return predictor, history


def test_predictor_returns_one_prediction_per_eligible_event(
    predictor_and_history,
):
    predictor, history = predictor_and_history

    results = predictor.predict(history)

    assert len(results) == history["event_id"].nunique()
    assert results["event_id"].is_unique
    assert set(results["event_id"]) == set(history["event_id"])


def test_predictor_respects_two_day_cutoff(
    predictor_and_history,
):
    predictor, history = predictor_and_history

    results = predictor.predict(history)

    assert (results["selected_time_to_tca_days"] >= 2.0).all()
    assert (results["prediction_cutoff_days"] == 2.0).all()


def test_predictor_rejects_non_dataframe_input(
    predictor_and_history,
):
    predictor, _ = predictor_and_history

    with pytest.raises(TypeError):
        predictor.predict([{"event_id": "A"}])


def test_predictor_rejects_missing_feature_columns(
    predictor_and_history,
):
    predictor, history = predictor_and_history

    with pytest.raises(ValueError, match="feature schema mismatch"):
        predictor.predict(
            history.drop(columns=["miss_distance"])
        )


def test_predictor_returns_unique_event_predictions(
    predictor_and_history,
):
    predictor, history = predictor_and_history

    duplicated = pd.concat(
        [history, history.iloc[[0]]],
        ignore_index=True,
    )

    results = predictor.predict(duplicated)

    assert results["event_id"].is_unique
    assert len(results) == history["event_id"].nunique()


def test_predictor_output_contains_research_status(
    predictor_and_history,
):
    predictor, history = predictor_and_history

    results = predictor.predict(history)

    assert (
        results["operational_status"] == "Research prototype"
    ).all()
    assert (
        results["prediction_target"]
        == "final_event_base10_log_risk"
    ).all()


def test_predictor_returns_finite_predictions(
    predictor_and_history,
):
    predictor, history = predictor_and_history

    results = predictor.predict(history)

    assert np.isfinite(
        results["predicted_final_risk"].to_numpy(dtype=float)
    ).all()


def test_classifier_floor_predictions_skip_regression(
    predictor_and_history,
):
    predictor, history = predictor_and_history

    results = predictor.predict(history)

    assert results["predicted_risk_floor"].iloc[0]
    assert results["predicted_final_risk"].iloc[0] == -30.0

    assert not results["predicted_risk_floor"].iloc[1:].any()
    assert (
        results["predicted_final_risk"].iloc[1:] == -10.0
    ).all()
