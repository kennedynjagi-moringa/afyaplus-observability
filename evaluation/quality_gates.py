"""
Predefined clinical safety thresholds for the evaluation quality gate.

These are deliberately strict because AfyaPlus answers can affect
triage/routing and medication decisions: a model that does not clear
these gates should not be trusted to serve that clinical feature
unsupervised.
"""

QUALITY_GATE_THRESHOLDS = {
    "rouge_l": 0.30,
    "token_f1": 0.45,
    "correctness": 4.0,
    "groundedness": 4.0,
    "overall_alignment": 4.0,
}


def evaluate_quality_gate(mean_scores: dict) -> dict:
    """
    mean_scores: dict with keys matching QUALITY_GATE_THRESHOLDS, holding
    the mean value of that metric for one (model, feature) group.

    Returns a dict of per-metric pass/fail plus an overall pass/fail
    (overall passes only if every individual gate passes).
    """
    result = {}
    for metric, threshold in QUALITY_GATE_THRESHOLDS.items():
        value = mean_scores.get(metric, 0.0)
        result[f"{metric}_pass"] = bool(value >= threshold)

    result["overall_pass"] = all(
        result[f"{metric}_pass"] for metric in QUALITY_GATE_THRESHOLDS
    )
    return result
