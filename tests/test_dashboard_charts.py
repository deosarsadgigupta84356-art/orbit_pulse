import pandas as pd

from dashboard.charts import (
    create_ablation_comparison_chart,
    create_risk_distribution_chart,
    create_precision_recall_chart,
    create_error_distribution_chart,
)


def test_missing_ablation_metrics_are_not_plotted_as_zero():
    fig = create_ablation_comparison_chart({
        "A_baseline_last_pc": {"metrics": {"mae_log10": 0.12}}
    })

    assert list(fig.data[0].y) == [0.12, None, None, None]
    assert list(fig.data[0].text) == ["0.1200", "N/A", "N/A", "N/A"]


def test_risk_distribution_handles_empty_event_table():
    fig = create_risk_distribution_chart(pd.DataFrame(columns=["risk_band"]))

    assert len(fig.data) == 1
    assert len(fig.layout.annotations) == 1


def test_risk_distribution_counts_each_band():
    summary = pd.DataFrame({"risk_band": ["HIGH", "LOW", "HIGH", "UNCERTAIN"]})
    fig = create_risk_distribution_chart(summary)

    assert list(fig.data[0].labels) == ["HIGH", "UNCERTAIN", "LOW"]
    assert list(fig.data[0].values) == [2, 1, 1]



def test_precision_recall_and_error_charts_accept_evaluation_payload():
    curve = {"precision": [1.0, 0.5], "recall": [0.0, 1.0], "thresholds": [-5.0]}
    pr_fig = create_precision_recall_chart(curve, curve)
    error_fig = create_error_distribution_chart([0.1, 0.2], [0.15, 0.25])

    assert len(pr_fig.data) == 2
    assert len(error_fig.data) == 2
