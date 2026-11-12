
import pandas as pd
import pytest

from src.esa_features import (
    build_esa_event_features,
    build_esa_training_dataset,
)


@pytest.fixture
def sample_esa_data():
    return pd.DataFrame({
        "event_id": [
            "A", "A", "A",
            "B",
            "C", "C", "C",
        ],
        "mission_id": [
            101, 101, 101,
            102,
            103, 103, 103,
        ],
        "time_to_tca": [
            5.0, 2.0, 1.0,
            1.5,
            6.0, 2.5, 0.5,
        ],
        "risk": [
            -10.0, -12.0, -30.0,
            -8.0,
            -7.0, -9.0, -5.0,
        ],
        "miss_distance": [
            15000.0, 12000.0, 10000.0,
            9000.0,
            20000.0, 18000.0, 15000.0,
        ],
        "c_object_type": [
            "DEBRIS", "DEBRIS", "DEBRIS",
            "PAYLOAD",
            "UNKNOWN", "UNKNOWN", "UNKNOWN",
        ],
    })


def test_latest_eligible_cdm_is_selected(sample_esa_data):
    X = build_esa_event_features(sample_esa_data)

    assert list(X.index) == ["A", "C"]
    assert X.loc["A", "time_to_tca"] == 2.0
    assert X.loc["C", "time_to_tca"] == 2.5


def test_two_day_cutoff_is_respected(sample_esa_data):
    X = build_esa_event_features(sample_esa_data)

    assert (X["time_to_tca"] >= 2.0).all()


def test_identifiers_are_not_model_features(sample_esa_data):
    X = build_esa_event_features(sample_esa_data)

    assert "event_id" not in X.columns
    assert "mission_id" not in X.columns
    assert X.index.name == "event_id"


def test_final_labels_use_complete_event_history(sample_esa_data):
    X, y = build_esa_training_dataset(sample_esa_data)

    assert y.loc["A", "final_risk"] == -30.0
    assert y.loc["C", "final_risk"] == -5.0


def test_events_without_eligible_records_are_excluded(sample_esa_data):
    X, y = build_esa_training_dataset(sample_esa_data)

    assert "B" not in X.index
    assert "B" not in y.index
    assert len(X) == 2


def test_risk_floor_classification(sample_esa_data):
    X, y = build_esa_training_dataset(sample_esa_data)

    assert bool(y.loc["A", "risk_floor"]) is True
    assert bool(y.loc["C", "risk_floor"]) is False


def test_feature_and_target_alignment(sample_esa_data):
    X, y = build_esa_training_dataset(sample_esa_data)

    assert X.index.equals(y.index)
    assert not X.index.has_duplicates


def test_invalid_cutoff_is_rejected(sample_esa_data):
    with pytest.raises(ValueError):
        build_esa_event_features(
            sample_esa_data,
            prediction_cutoff_days=0,
        )


def test_no_eligible_events_is_rejected(sample_esa_data):
    with pytest.raises(ValueError):
        build_esa_event_features(
            sample_esa_data,
            prediction_cutoff_days=10,
        )
