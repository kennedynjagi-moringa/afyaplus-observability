"""
Reads the CSV/HTML outputs produced by Phases 1-3 and shapes them into
the 4 dashboard sections. Every function is defensive: if an output file
is missing (e.g. a phase hasn't been run yet), it reports that clearly
instead of crashing the dashboard, and counts it as an exception for the
System Health section / Prometheus counter.

This module also imports common.llm_client, whose import fails immediately
if OPENAI_API_KEY is missing.
"""

import os
import time

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVALUATION_DIR = os.path.join(ROOT_DIR, "evaluation")
DRIFT_DIR = os.path.join(ROOT_DIR, "drift")
COST_DIR = os.path.join(ROOT_DIR, "cost")

from dashboard.config import DAILY_BUDGET_CAP_USD, MONTHLY_BUDGET_CAP_USD  # noqa: E402

_exception_count = 0


def _record_exception() -> None:
    global _exception_count
    _exception_count += 1


def _read_csv(path: str) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
        _record_exception()
        return None


def get_system_health() -> dict:
    checks = {}

    checks["OPENAI_API_KEY configured"] = False
    try:
        from common.llm_client import get_client  # noqa: F401

        checks["OPENAI_API_KEY configured"] = True
    except ValueError:
        _record_exception()

    required_outputs = {
        "Phase 1 evaluation results": os.path.join(EVALUATION_DIR, "full_evaluation_results.csv"),
        "Phase 2 drift trend table": os.path.join(DRIFT_DIR, "drift_trend_table.csv"),
        "Phase 3 cost projection": os.path.join(COST_DIR, "cost_projection_30day.csv"),
    }
    for label, path in required_outputs.items():
        checks[label] = os.path.isfile(path)
        if not checks[label]:
            _record_exception()

    status = "UP" if all(checks.values()) else "DOWN"

    return {
        "status": status,
        "checks": checks,
        "exception_count": _exception_count,
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }


def get_quality_matrix() -> list[dict]:
    agg = _read_csv(os.path.join(EVALUATION_DIR, "model_comparison_aggregation.csv"))
    gate = _read_csv(os.path.join(EVALUATION_DIR, "quality_gate_log.csv"))
    savings = _read_csv(os.path.join(COST_DIR, "structural_savings_analysis.csv"))

    if agg is None or gate is None:
        return []

    routing_by_feature = {}
    if savings is not None:
        routing_by_feature = dict(zip(savings["feature"], savings["optimal_model"]))

    gate_lookup = {
        (row["model"], row["feature"]): bool(row["overall_pass"]) for _, row in gate.iterrows()
    }

    rows = []
    for _, row in agg.iterrows():
        key = (row["model"], row["feature"])
        rows.append(
            {
                "feature": row["feature"],
                "model": row["model"],
                "overall_alignment": round(row["overall_alignment"], 2),
                "correctness": round(row["correctness"], 2),
                "groundedness": round(row["groundedness"], 2),
                "rouge_l": round(row["rouge_l"], 3),
                "token_f1": round(row["token_f1"], 3),
                "quality_gate_pass": gate_lookup.get(key, False),
                "recommended_routing": routing_by_feature.get(row["feature"], "n/a"),
            }
        )
    return rows


def get_drift_status() -> dict:
    trend = _read_csv(os.path.join(DRIFT_DIR, "drift_trend_table.csv"))
    alerts = _read_csv(os.path.join(DRIFT_DIR, "drift_alert_log.csv"))

    if trend is None or alerts is None:
        return {"current_month": None, "columns": []}

    current_month = int(trend["month"].max())
    baseline = trend[trend["month"] == 1].iloc[0]
    current = trend[trend["month"] == current_month].iloc[0]
    current_alerts = alerts[alerts["month"] == current_month].set_index("column")

    columns = []
    for column in ["rouge_l", "latency_ms", "input_token_length"]:
        alert_row = current_alerts.loc[column] if column in current_alerts.index else None
        columns.append(
            {
                "column": column,
                "baseline_month1_mean": round(float(baseline[column]), 4),
                "current_mean": round(float(current[column]), 4),
                "p_value": round(float(alert_row["p_value"]), 6) if alert_row is not None else None,
                "drifted": bool(alert_row["drifted"]) if alert_row is not None else False,
            }
        )

    return {"current_month": current_month, "columns": columns}


def get_budget_status() -> dict:
    projection = _read_csv(os.path.join(COST_DIR, "cost_projection_30day.csv"))
    if projection is None:
        return {}

    monthly_spend = float(projection["cost_usd"].sum())
    last_day = int(projection["day"].max())
    daily_spend = float(projection[projection["day"] == last_day]["cost_usd"].sum())

    return {
        "daily_spend_usd": round(daily_spend, 4),
        "daily_cap_usd": DAILY_BUDGET_CAP_USD,
        "daily_utilization": round(min(daily_spend / DAILY_BUDGET_CAP_USD, 1.5), 3),
        "monthly_spend_usd": round(monthly_spend, 2),
        "monthly_cap_usd": MONTHLY_BUDGET_CAP_USD,
        "monthly_utilization": round(min(monthly_spend / MONTHLY_BUDGET_CAP_USD, 1.5), 3),
    }
