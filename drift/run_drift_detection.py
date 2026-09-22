"""
Phase 2: Statistical Drift Detection.

Simulates 3 months of AfyaPlus production traffic (drift/traffic_simulator.py),
then uses Evidently AI to compare each month's rouge_l / latency_ms /
input_token_length distributions against the Month-1 baseline.

Outputs (all written into drift/):
  - simulated_traffic.csv          (raw per-request simulated data, all months)
  - month1_drift_report.html       (baseline internal-consistency check)
  - month2_drift_report.html       (month 2 vs month 1 baseline)
  - month3_drift_report.html       (month 3 vs month 1 baseline)
  - drift_trend_table.csv          (monthly means of the 3 tracked columns)
  - drift_alert_log.csv            (machine-readable: which month/column first drifted)

Run from the project root:
    .venv/bin/python -m drift.run_drift_detection
"""

import os

import pandas as pd
from evidently import Dataset, Report
from evidently.presets import DataDriftPreset

from drift.traffic_simulator import simulate_month

OUTPUT_DIR = os.path.dirname(__file__)
TRACKED_COLUMNS = ["rouge_l", "latency_ms", "input_token_length"]
DRIFT_P_VALUE_THRESHOLD = 0.05


def extract_value_drift(snapshot) -> dict:
    """column -> K-S test p-value, from an Evidently DataDriftPreset snapshot."""
    p_values = {}
    for metric in snapshot.dict()["metrics"]:
        config = metric.get("config", {})
        if config.get("type") == "evidently:metric_v2:ValueDrift":
            p_values[config["column"]] = float(metric["value"])
    return p_values


def run_report(current_df: pd.DataFrame, reference_df: pd.DataFrame, html_path: str) -> dict:
    current_ds = Dataset.from_pandas(current_df[TRACKED_COLUMNS])
    reference_ds = Dataset.from_pandas(reference_df[TRACKED_COLUMNS])

    report = Report([DataDriftPreset(columns=TRACKED_COLUMNS)])
    snapshot = report.run(current_ds, reference_ds)
    snapshot.save_html(html_path)
    print(f"Saved drift report -> {html_path}")

    return extract_value_drift(snapshot)


def run() -> None:
    all_rows = []
    for month in (1, 2, 3):
        print(f"Simulating month {month} traffic...")
        all_rows.extend(simulate_month(month, seed=1000 + month))

    traffic_df = pd.DataFrame(all_rows)
    traffic_path = os.path.join(OUTPUT_DIR, "simulated_traffic.csv")
    traffic_df.to_csv(traffic_path, index=False)
    print(f"Saved simulated traffic -> {traffic_path}")

    month1 = traffic_df[traffic_df["month"] == 1].reset_index(drop=True)
    month2 = traffic_df[traffic_df["month"] == 2].reset_index(drop=True)
    month3 = traffic_df[traffic_df["month"] == 3].reset_index(drop=True)

    # Month 1 report: internal baseline consistency check (first half vs second half).
    half = len(month1) // 2
    run_report(
        current_df=month1.iloc[half:],
        reference_df=month1.iloc[:half],
        html_path=os.path.join(OUTPUT_DIR, "month1_drift_report.html"),
    )

    month2_p_values = run_report(
        current_df=month2,
        reference_df=month1,
        html_path=os.path.join(OUTPUT_DIR, "month2_drift_report.html"),
    )
    month3_p_values = run_report(
        current_df=month3,
        reference_df=month1,
        html_path=os.path.join(OUTPUT_DIR, "month3_drift_report.html"),
    )

    # ---- Drift trend table ----
    trend = (
        traffic_df.groupby("month")[TRACKED_COLUMNS]
        .mean()
        .round(4)
        .reset_index()
    )
    trend_path = os.path.join(OUTPUT_DIR, "drift_trend_table.csv")
    trend.to_csv(trend_path, index=False)
    print(f"Saved drift trend table -> {trend_path}")

    # ---- Automated alert log ----
    alert_rows = []
    for month, p_values in ((2, month2_p_values), (3, month3_p_values)):
        for column in TRACKED_COLUMNS:
            p_value = p_values.get(column)
            drifted = p_value is not None and p_value < DRIFT_P_VALUE_THRESHOLD
            alert_rows.append(
                {
                    "month": month,
                    "column": column,
                    "method": "K-S p_value",
                    "p_value": p_value,
                    "threshold": DRIFT_P_VALUE_THRESHOLD,
                    "drifted": drifted,
                }
            )
    alert_df = pd.DataFrame(alert_rows)

    first_drift_rows = []
    for column in TRACKED_COLUMNS:
        col_alerts = alert_df[(alert_df["column"] == column) & (alert_df["drifted"])]
        first_month = int(col_alerts["month"].min()) if not col_alerts.empty else None
        first_drift_rows.append({"column": column, "first_drifted_month": first_month})
    first_drift_df = pd.DataFrame(first_drift_rows)

    alert_path = os.path.join(OUTPUT_DIR, "drift_alert_log.csv")
    alert_df.to_csv(alert_path, index=False)
    print(f"Saved drift alert log -> {alert_path}")

    print("\n=== First month each column drifted (p < 0.05 vs Month-1 baseline) ===")
    print(first_drift_df.to_string(index=False))


if __name__ == "__main__":
    run()
