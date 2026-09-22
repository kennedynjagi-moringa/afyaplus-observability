from evaluation.quality_gates import QUALITY_GATE_THRESHOLDS, evaluate_quality_gate


def test_all_metrics_above_threshold_passes():
    scores = {metric: threshold + 0.5 for metric, threshold in QUALITY_GATE_THRESHOLDS.items()}
    result = evaluate_quality_gate(scores)
    assert result["overall_pass"] is True
    assert all(result[f"{m}_pass"] for m in QUALITY_GATE_THRESHOLDS)


def test_one_metric_below_threshold_fails_overall():
    scores = {metric: threshold + 0.5 for metric, threshold in QUALITY_GATE_THRESHOLDS.items()}
    scores["token_f1"] = 0.0
    result = evaluate_quality_gate(scores)
    assert result["token_f1_pass"] is False
    assert result["overall_pass"] is False


def test_missing_metric_treated_as_zero_and_fails():
    result = evaluate_quality_gate({})
    assert result["overall_pass"] is False
